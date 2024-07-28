/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




class sipFbxLayerElementArrayTemplate_FbxColor : public  ::FbxLayerElementArrayTemplate_FbxColor
{
public:
    sipFbxLayerElementArrayTemplate_FbxColor( ::EFbxType);
    sipFbxLayerElementArrayTemplate_FbxColor(const  ::FbxLayerElementArrayTemplate_FbxColor&);
    virtual ~sipFbxLayerElementArrayTemplate_FbxColor();

    /*
     * There is a protected method for every virtual method visible from
     * this class.
     */
protected:
    void* GetLocked( ::FbxLayerElementArray::ELockMode, ::EFbxType);
    void Release(void**, ::EFbxType);
     ::size_t GetStride() const;

public:
    sipSimpleWrapper *sipPySelf;

private:
    sipFbxLayerElementArrayTemplate_FbxColor(const sipFbxLayerElementArrayTemplate_FbxColor &);
    sipFbxLayerElementArrayTemplate_FbxColor &operator = (const sipFbxLayerElementArrayTemplate_FbxColor &);

    char sipPyMethods[3];
};

sipFbxLayerElementArrayTemplate_FbxColor::sipFbxLayerElementArrayTemplate_FbxColor( ::EFbxType a0):  ::FbxLayerElementArrayTemplate_FbxColor(a0), sipPySelf(0)
{
    memset(sipPyMethods, 0, sizeof (sipPyMethods));
}

sipFbxLayerElementArrayTemplate_FbxColor::sipFbxLayerElementArrayTemplate_FbxColor(const  ::FbxLayerElementArrayTemplate_FbxColor& a0):  ::FbxLayerElementArrayTemplate_FbxColor(a0), sipPySelf(0)
{
    memset(sipPyMethods, 0, sizeof (sipPyMethods));
}

sipFbxLayerElementArrayTemplate_FbxColor::~sipFbxLayerElementArrayTemplate_FbxColor()
{
    sipInstanceDestroyed(sipPySelf);
}

void* sipFbxLayerElementArrayTemplate_FbxColor::GetLocked( ::FbxLayerElementArray::ELockMode a0, ::EFbxType a1)
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[0],sipPySelf,NULL,sipName_GetLocked);

    if (!sipMeth)
        return  ::FbxLayerElementArrayTemplate_FbxColor::GetLocked(a0,a1);

    extern void* sipVH_fbx_7(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *,  ::FbxLayerElementArray::ELockMode, ::EFbxType);

    return sipVH_fbx_7(sipGILState, 0, sipPySelf, sipMeth, a0, a1);
}

void sipFbxLayerElementArrayTemplate_FbxColor::Release(void**a0, ::EFbxType a1)
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[1],sipPySelf,NULL,sipName_Release);

    if (!sipMeth)
    {
         ::FbxLayerElementArrayTemplate_FbxColor::Release(a0,a1);
        return;
    }

    extern void sipVH_fbx_8(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *, void**, ::EFbxType);

    sipVH_fbx_8(sipGILState, 0, sipPySelf, sipMeth, a0, a1);
}

 ::size_t sipFbxLayerElementArrayTemplate_FbxColor::GetStride() const
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,const_cast<char *>(&sipPyMethods[2]),sipPySelf,NULL,sipName_GetStride);

    if (!sipMeth)
        return  ::FbxLayerElementArrayTemplate_FbxColor::GetStride();

    extern  ::size_t sipVH_fbx_9(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *);

    return sipVH_fbx_9(sipGILState, 0, sipPySelf, sipMeth);
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_Add, "Add(self, FbxColor) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_Add(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_Add(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxColor* a0;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, sipType_FbxColor, &a0))
        {
            int sipRes;

            sipRes = sipCpp->Add(*a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_Add, doc_FbxLayerElementArrayTemplate_FbxColor_Add);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_InsertAt, "InsertAt(self, int, FbxColor) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_InsertAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_InsertAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxColor* a1;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0, sipType_FbxColor, &a1))
        {
            int sipRes;

            sipRes = sipCpp->InsertAt(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_InsertAt, doc_FbxLayerElementArrayTemplate_FbxColor_InsertAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_SetAt, "SetAt(self, int, FbxColor)");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_SetAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_SetAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxColor* a1;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0, sipType_FbxColor, &a1))
        {
            sipCpp->SetAt(a0,*a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_SetAt, doc_FbxLayerElementArrayTemplate_FbxColor_SetAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_SetLast, "SetLast(self, FbxColor)");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_SetLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_SetLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxColor* a0;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, sipType_FbxColor, &a0))
        {
            sipCpp->SetLast(*a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_SetLast, doc_FbxLayerElementArrayTemplate_FbxColor_SetLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveAt, "RemoveAt(self, int) -> FbxColor");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0))
        {
             ::FbxColor*sipRes;

            sipRes = new  ::FbxColor(sipCpp->RemoveAt(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_RemoveAt, doc_FbxLayerElementArrayTemplate_FbxColor_RemoveAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveLast, "RemoveLast(self) -> FbxColor");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp))
        {
             ::FbxColor*sipRes;

            sipRes = new  ::FbxColor(sipCpp->RemoveLast());

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_RemoveLast, doc_FbxLayerElementArrayTemplate_FbxColor_RemoveLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveIt, "RemoveIt(self, FbxColor) -> bool");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveIt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_RemoveIt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxColor* a0;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, sipType_FbxColor, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->RemoveIt(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_RemoveIt, doc_FbxLayerElementArrayTemplate_FbxColor_RemoveIt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_GetAt, "GetAt(self, int) -> FbxColor");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        const  ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0))
        {
             ::FbxColor*sipRes;

            sipRes = new  ::FbxColor(sipCpp->GetAt(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_GetAt, doc_FbxLayerElementArrayTemplate_FbxColor_GetAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_GetFirst, "GetFirst(self) -> FbxColor");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetFirst(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetFirst(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp))
        {
             ::FbxColor*sipRes;

            sipRes = new  ::FbxColor(sipCpp->GetFirst());

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_GetFirst, doc_FbxLayerElementArrayTemplate_FbxColor_GetFirst);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_GetLast, "GetLast(self) -> FbxColor");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_GetLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp))
        {
             ::FbxColor*sipRes;

            sipRes = new  ::FbxColor(sipCpp->GetLast());

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_GetLast, doc_FbxLayerElementArrayTemplate_FbxColor_GetLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_Find, "Find(self, FbxColor) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_Find(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_Find(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxColor* a0;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, sipType_FbxColor, &a0))
        {
            int sipRes;

            sipRes = sipCpp->Find(*a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_Find, doc_FbxLayerElementArrayTemplate_FbxColor_Find);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_FindAfter, "FindAfter(self, int, FbxColor) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_FindAfter(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_FindAfter(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxColor* a1;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0, sipType_FbxColor, &a1))
        {
            int sipRes;

            sipRes = sipCpp->FindAfter(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_FindAfter, doc_FbxLayerElementArrayTemplate_FbxColor_FindAfter);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor_FindBefore, "FindBefore(self, int, FbxColor) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_FindBefore(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxColor_FindBefore(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxColor* a1;
         ::FbxLayerElementArrayTemplate_FbxColor *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxColor, &sipCpp, &a0, sipType_FbxColor, &a1))
        {
            int sipRes;

            sipRes = sipCpp->FindBefore(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName_FindBefore, doc_FbxLayerElementArrayTemplate_FbxColor_FindBefore);

    return NULL;
}


extern "C" {static PyObject *slot_FbxLayerElementArrayTemplate_FbxColor___getitem__(PyObject *,PyObject *);}
static PyObject *slot_FbxLayerElementArrayTemplate_FbxColor___getitem__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxLayerElementArrayTemplate_FbxColor *sipCpp = reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxColor *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxLayerElementArrayTemplate_FbxColor));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        int a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1i", &a0))
        {
             ::FbxColor*sipRes = 0;
            int sipIsErr = 0;

#line 290 "/home/ke/Documents/expressive-humanoid/sip/fbxlayerelementarray.sip"
        if (a0 < 0 || a0 >= sipCpp->GetCount())
        {
            PyErr_Format(PyExc_IndexError, "sequence index out of range");
            sipIsErr = 1;
        }
        else
        {
            fbxArrayElementCopy(&sipRes, (FbxColor*)NULL, sipCpp, a0);
        }
#line 538 "/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipfbxFbxLayerElementArrayTemplate_FbxColor.cpp"

            if (sipIsErr)
                return 0;

            return sipConvertFromNewType(sipRes,sipType_FbxColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxColor, sipName___getitem__, NULL);

    return 0;
}


/* Cast a pointer to a type somewhere in its inheritance hierarchy. */
extern "C" {static void *cast_FbxLayerElementArrayTemplate_FbxColor(void *, const sipTypeDef *);}
static void *cast_FbxLayerElementArrayTemplate_FbxColor(void *sipCppV, const sipTypeDef *targetType)
{
     ::FbxLayerElementArrayTemplate_FbxColor *sipCpp = reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxColor *>(sipCppV);

    if (targetType == sipType_FbxLayerElementArray)
        return static_cast< ::FbxLayerElementArray *>(sipCpp);

    return sipCppV;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxLayerElementArrayTemplate_FbxColor(void *, int);}
static void release_FbxLayerElementArrayTemplate_FbxColor(void *sipCppV, int sipState)
{
    if (sipState & SIP_DERIVED_CLASS)
        delete reinterpret_cast<sipFbxLayerElementArrayTemplate_FbxColor *>(sipCppV);
    else
        delete reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxColor *>(sipCppV);
}


extern "C" {static void dealloc_FbxLayerElementArrayTemplate_FbxColor(sipSimpleWrapper *);}
static void dealloc_FbxLayerElementArrayTemplate_FbxColor(sipSimpleWrapper *sipSelf)
{
    if (sipIsDerivedClass(sipSelf))
        reinterpret_cast<sipFbxLayerElementArrayTemplate_FbxColor *>(sipGetAddress(sipSelf))->sipPySelf = NULL;

    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxLayerElementArrayTemplate_FbxColor(sipGetAddress(sipSelf), sipIsDerivedClass(sipSelf));
    }
}


extern "C" {static void *init_type_FbxLayerElementArrayTemplate_FbxColor(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxLayerElementArrayTemplate_FbxColor(sipSimpleWrapper *sipSelf, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
    sipFbxLayerElementArrayTemplate_FbxColor *sipCpp = 0;

    {
         ::EFbxType a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "E", sipType_EFbxType, &a0))
        {
            sipCpp = new sipFbxLayerElementArrayTemplate_FbxColor(a0);

            sipCpp->sipPySelf = sipSelf;

            return sipCpp;
        }
    }

    {
        const  ::FbxLayerElementArrayTemplate_FbxColor* a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "J9", sipType_FbxLayerElementArrayTemplate_FbxColor, &a0))
        {
            sipCpp = new sipFbxLayerElementArrayTemplate_FbxColor(*a0);

            sipCpp->sipPySelf = sipSelf;

            return sipCpp;
        }
    }

    return NULL;
}


/* Define this type's super-types. */
static sipEncodedTypeDef supers_FbxLayerElementArrayTemplate_FbxColor[] = {{176, 255, 1}};


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxLayerElementArrayTemplate_FbxColor[] = {
    {(void *)slot_FbxLayerElementArrayTemplate_FbxColor___getitem__, getitem_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxLayerElementArrayTemplate_FbxColor[] = {
    {SIP_MLNAME_CAST(sipName_Add), meth_FbxLayerElementArrayTemplate_FbxColor_Add, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_Add)},
    {SIP_MLNAME_CAST(sipName_Find), meth_FbxLayerElementArrayTemplate_FbxColor_Find, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_Find)},
    {SIP_MLNAME_CAST(sipName_FindAfter), meth_FbxLayerElementArrayTemplate_FbxColor_FindAfter, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_FindAfter)},
    {SIP_MLNAME_CAST(sipName_FindBefore), meth_FbxLayerElementArrayTemplate_FbxColor_FindBefore, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_FindBefore)},
    {SIP_MLNAME_CAST(sipName_GetAt), meth_FbxLayerElementArrayTemplate_FbxColor_GetAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_GetAt)},
    {SIP_MLNAME_CAST(sipName_GetFirst), meth_FbxLayerElementArrayTemplate_FbxColor_GetFirst, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_GetFirst)},
    {SIP_MLNAME_CAST(sipName_GetLast), meth_FbxLayerElementArrayTemplate_FbxColor_GetLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_GetLast)},
    {SIP_MLNAME_CAST(sipName_InsertAt), meth_FbxLayerElementArrayTemplate_FbxColor_InsertAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_InsertAt)},
    {SIP_MLNAME_CAST(sipName_RemoveAt), meth_FbxLayerElementArrayTemplate_FbxColor_RemoveAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveAt)},
    {SIP_MLNAME_CAST(sipName_RemoveIt), meth_FbxLayerElementArrayTemplate_FbxColor_RemoveIt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveIt)},
    {SIP_MLNAME_CAST(sipName_RemoveLast), meth_FbxLayerElementArrayTemplate_FbxColor_RemoveLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_RemoveLast)},
    {SIP_MLNAME_CAST(sipName_SetAt), meth_FbxLayerElementArrayTemplate_FbxColor_SetAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_SetAt)},
    {SIP_MLNAME_CAST(sipName_SetLast), meth_FbxLayerElementArrayTemplate_FbxColor_SetLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxColor_SetLast)}
};

PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxColor, "\1FbxLayerElementArrayTemplate_FbxColor(EFbxType)\n"
    "FbxLayerElementArrayTemplate_FbxColor(FbxLayerElementArrayTemplate_FbxColor)");


sipClassTypeDef sipTypeDef_fbx_FbxLayerElementArrayTemplate_FbxColor = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxLayerElementArrayTemplate_FbxColor,
        {0},
        0
    },
    {
        sipNameNr_FbxLayerElementArrayTemplate_FbxColor,
        {0, 0, 1},
        13, methods_FbxLayerElementArrayTemplate_FbxColor,
        0, 0,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxLayerElementArrayTemplate_FbxColor,
    -1,
    -1,
    supers_FbxLayerElementArrayTemplate_FbxColor,
    slots_FbxLayerElementArrayTemplate_FbxColor,
    init_type_FbxLayerElementArrayTemplate_FbxColor,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxLayerElementArrayTemplate_FbxColor,
    0,
    0,
    0,
    release_FbxLayerElementArrayTemplate_FbxColor,
    cast_FbxLayerElementArrayTemplate_FbxColor,
    0,
    0,
    0,
    0,
    0,
    0
};
